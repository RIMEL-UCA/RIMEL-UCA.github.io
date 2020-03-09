package fr.unice.polytech.si5.rimel.eventprofile.repository;

import fr.unice.polytech.si5.rimel.eventprofile.model.MavenProfile;
import org.springframework.data.neo4j.annotation.Query;
import org.springframework.data.neo4j.repository.Neo4jRepository;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

@Repository
public interface IProfileRepository extends Neo4jRepository<MavenProfile, String> {

	Optional<MavenProfile> findByName(String name);

}
